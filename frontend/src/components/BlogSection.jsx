import BlogCard from "./BlogCard"

const data = [
    {
        img: "/blog_1.jpg",
        title: "Digital Agriculture",
        date: "Mar 23, 2024",
        comment: 8,
    },
    {
        img: "/blog_2.jpg",
        title: "Transforming Indian Agriculture",
        date: "Feb 2, 2024",
        comment: 10,
    },
    {
        img: "/blog_3.png",
        title: "Precision Agriculture and its challenges",
        date: "Jan 15, 2024",
        comment: 5,
    },
]

const BlogSection = () => {
  return (
    <div className="container p-16">
        <h2 className="font-bold text-2xl">Latest News</h2>
        <p className="text-gray-500">
            Interesting blogs related to Agriculture.
        </p>

        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-8 pt-8">
            {data.map((el) => (
                <BlogCard 
                    key={el.date}
                    img={el.img}
                    title={el.title}
                    date={el.date}
                    comment={el.comment}
                />
            ))}
        </div>
    </div>
  )
}

export default BlogSection
